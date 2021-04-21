import web3 from './web3';
import { modelDatabase} from "../abi/abi";

const address = '0xC8f0F368321b6e9403a26818317F5812d4A64639';

export default new web3.eth.Contract(modelDatabase, address);
