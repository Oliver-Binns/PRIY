/*
* 
*/
package FLAME.diagram.part;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.eclipse.core.commands.ExecutionException;
import org.eclipse.core.resources.IFile;
import org.eclipse.core.resources.IResource;
import org.eclipse.core.resources.IResourceStatus;
import org.eclipse.core.resources.IStorage;
import org.eclipse.core.resources.ResourcesPlugin;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.IAdaptable;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.IStatus;
import org.eclipse.core.runtime.Path;
import org.eclipse.core.runtime.Status;
import org.eclipse.core.runtime.jobs.ISchedulingRule;
import org.eclipse.core.runtime.jobs.MultiRule;
import org.eclipse.emf.common.notify.Adapter;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.Notifier;
import org.eclipse.emf.common.ui.URIEditorInput;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.util.EContentAdapter;
import org.eclipse.emf.ecore.util.EcoreUtil;
import org.eclipse.emf.transaction.NotificationFilter;
import org.eclipse.emf.transaction.TransactionalEditingDomain;
import org.eclipse.emf.workspace.util.WorkspaceSynchronizer;
import org.eclipse.gmf.runtime.common.core.command.CommandResult;
import org.eclipse.gmf.runtime.diagram.core.DiagramEditingDomainFactory;
import org.eclipse.gmf.runtime.diagram.ui.resources.editor.document.AbstractDocumentProvider;
import org.eclipse.gmf.runtime.diagram.ui.resources.editor.document.DiagramDocument;
import org.eclipse.gmf.runtime.diagram.ui.resources.editor.document.IDiagramDocument;
import org.eclipse.gmf.runtime.diagram.ui.resources.editor.document.IDiagramDocumentProvider;
import org.eclipse.gmf.runtime.diagram.ui.resources.editor.document.IDocument;
import org.eclipse.gmf.runtime.diagram.ui.resources.editor.internal.EditorStatusCodes;
import org.eclipse.gmf.runtime.diagram.ui.resources.editor.internal.util.DiagramIOUtil;
import org.eclipse.gmf.runtime.emf.commands.core.command.AbstractTransactionalCommand;
import org.eclipse.gmf.runtime.emf.core.resources.GMFResourceFactory;
import org.eclipse.gmf.runtime.notation.Diagram;
import org.eclipse.gmf.runtime.notation.View;
import org.eclipse.jface.operation.IRunnableContext;
import org.eclipse.osgi.util.NLS;
import org.eclipse.swt.widgets.Display;
import org.eclipse.ui.IEditorInput;
import org.eclipse.ui.part.FileEditorInput;

/**
 * @generated
 */
public class FLAMEDocumentProvider extends AbstractDocumentProvider implements IDiagramDocumentProvider {

	/**
	* @generated NOT
	*/
	protected void doSaveDocument(IProgressMonitor monitor, Object element, IDocument document, boolean overwrite)
			throws CoreException {
		ValidateAction.runValidation((View) document.getContent());
		ResourceSetInfo info = getResourceSetInfo(element);
		if (info != null) {
			if (!overwrite && !info.isSynchronized()) {
				throw new CoreException(
						new Status(IStatus.ERROR, FLAMEDiagramEditorPlugin.ID, IResourceStatus.OUT_OF_SYNC_LOCAL,
								Messages.FLAMEDocumentProvider_UnsynchronizedFileSaveError, null));
			}
			info.stopResourceListening();
			fireElementStateChanging(element);
			try {
				monitor.beginTask(Messages.FLAMEDocumentProvider_SaveDiagramTask,
						info.getResourceSet().getResources().size() + 1); //"Saving diagram"
				for (Iterator<Resource> it = info.getLoadedResourcesIterator(); it.hasNext();) {
					Resource nextResource = it.next();
					monitor.setTaskName(
							NLS.bind(Messages.FLAMEDocumentProvider_SaveNextResourceTask, nextResource.getURI()));
					if (nextResource.isLoaded() && !info.getEditingDomain().isReadOnly(nextResource)) {
						try {
							nextResource.save(FLAMEDiagramEditorUtil.getSaveOptions());
						} catch (IOException e) {
							fireElementStateChangeFailed(element);
							throw new CoreException(new Status(IStatus.ERROR, FLAMEDiagramEditorPlugin.ID,
									EditorStatusCodes.RESOURCE_FAILURE, e.getLocalizedMessage(), null));
						}
					}
					monitor.worked(1);
				}
				monitor.done();
				info.setModificationStamp(computeModificationStamp(info));
			} catch (RuntimeException x) {
				fireElementStateChangeFailed(element);
				throw x;
			} finally {
				info.startResourceListening();
			}
		} else {
			URI newResoruceURI;
			List<IFile> affectedFiles = null;
			if (element instanceof FileEditorInput) {
				IFile newFile = ((FileEditorInput) element).getFile();
				affectedFiles = Collections.singletonList(newFile);
				newResoruceURI = URI.createPlatformResourceURI(newFile.getFullPath().toString(), true);
			} else if (element instanceof URIEditorInput) {
				newResoruceURI = ((URIEditorInput) element).getURI();
			} else {
				fireElementStateChangeFailed(element);
				throw new CoreException(new Status(IStatus.ERROR, FLAMEDiagramEditorPlugin.ID, 0,
						NLS.bind(Messages.FLAMEDocumentProvider_IncorrectInputError,
								new Object[] { element, "org.eclipse.ui.part.FileEditorInput", //$NON-NLS-1$
										"org.eclipse.emf.common.ui.URIEditorInput" }), //$NON-NLS-1$ 
						null));
			}
			if (false == document instanceof IDiagramDocument) {
				fireElementStateChangeFailed(element);
				throw new CoreException(new Status(IStatus.ERROR, FLAMEDiagramEditorPlugin.ID, 0,
						"Incorrect document used: " + document //$NON-NLS-1$
								+ " instead of org.eclipse.gmf.runtime.diagram.ui.resources.editor.document.IDiagramDocument", //$NON-NLS-1$
						null));
			}
			IDiagramDocument diagramDocument = (IDiagramDocument) document;
			final Resource newResource = diagramDocument.getEditingDomain().getResourceSet()
					.createResource(newResoruceURI);
			final Diagram diagramCopy = (Diagram) EcoreUtil.copy(diagramDocument.getDiagram());
			try {
				new AbstractTransactionalCommand(diagramDocument.getEditingDomain(),
						NLS.bind(Messages.FLAMEDocumentProvider_SaveAsOperation, diagramCopy.getName()),
						affectedFiles) {
					protected CommandResult doExecuteWithResult(IProgressMonitor monitor, IAdaptable info)
							throws ExecutionException {
						newResource.getContents().add(diagramCopy);
						return CommandResult.newOKCommandResult();
					}
				}.execute(monitor, null);
				newResource.save(FLAMEDiagramEditorUtil.getSaveOptions());
			} catch (ExecutionException e) {
				fireElementStateChangeFailed(element);
				throw new CoreException(
						new Status(IStatus.ERROR, FLAMEDiagramEditorPlugin.ID, 0, e.getLocalizedMessage(), null));
			} catch (IOException e) {
				fireElementStateChangeFailed(element);
				throw new CoreException(
						new Status(IStatus.ERROR, FLAMEDiagramEditorPlugin.ID, 0, e.getLocalizedMessage(), null));
			}
			newResource.unload();
		}
	}
}
